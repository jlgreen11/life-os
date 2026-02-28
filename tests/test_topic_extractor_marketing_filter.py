"""
Tests for marketing email filtering in TopicExtractor.

The topic extractor must skip marketing/automated inbound emails so that
promotional vocabulary (offer, shop, holiday, rewards, deals, etc.) does not
flood the topic profile and produce garbage semantic facts like
``expertise_email`` or ``interest_shop``.

Verifies:
- Marketing inbound emails are skipped (no topics extracted)
- Genuine human inbound emails are processed
- Outbound emails are always processed (never automated)
- Direct messages (Signal/iMessage) are always processed
- User commands are always processed
"""

from __future__ import annotations

import pytest

from services.signal_extractor.topic import TopicExtractor


# ---------------------------------------------------------------------------
# Fixture: TopicExtractor instance with a minimal in-memory UMS stub
# ---------------------------------------------------------------------------

class _MinimalUMS:
    """Minimal stub that satisfies TopicExtractor's UMS calls during tests.

    TopicExtractor._update_topic_map() calls ums.get_signal_profile() and
    ums.update_signal_profile(), so we stub those methods without needing a
    real database.
    """

    def get_signal_profile(self, name: str):
        """Return empty profile data — no pre-existing topics."""
        return {"data": {"topic_counts": {}, "recent_topics": []}, "samples_count": 0}

    def update_signal_profile(self, name: str, data: dict):
        """No-op for testing — we don't need persistence here."""


class _MinimalDB:
    """Minimal DB stub — TopicExtractor's base class requires a db arg."""


@pytest.fixture
def topic_extractor():
    """Return a TopicExtractor wired to no-op in-memory stubs."""
    extractor = TopicExtractor(_MinimalDB(), _MinimalUMS())
    return extractor


# ---------------------------------------------------------------------------
# Marketing email events — should return []
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("from_addr", [
    "noreply@amazon.com",
    "no-reply@newsletter.example.com",
    "marketing@promo.company.com",
    "deals@shop.retailer.com",
    "notifications@accounts.google.com",
    "customerservice@example-store.com",
    "newsletter@mail.company.com",
])
def test_marketing_inbound_email_skipped(topic_extractor, from_addr):
    """Marketing inbound emails must not produce any topic signals."""
    event = {
        "type": "email.received",
        "timestamp": "2026-02-28T10:00:00Z",
        "payload": {
            "from": from_addr,
            "subject": "Exclusive holiday deals — shop now and save!",
            "body": (
                "Enjoy exclusive holiday offers and rewards this season. "
                "Shop our best deals and save big on gifts. "
                "Limited time only — visit our store today!"
            ),
        },
    }
    result = topic_extractor.extract(event)
    assert result == [], (
        f"Marketing email from {from_addr!r} should return [] but got {result}"
    )


def test_noreply_newsletter_skipped(topic_extractor):
    """Emails from noreply@newsletter.* senders must be skipped."""
    event = {
        "type": "email.received",
        "timestamp": "2026-02-28T10:00:00Z",
        "payload": {
            "from": "noreply@newsletter.example.com",
            "subject": "Your weekly digest",
            "body": "Top stories this week: technology, science, sports...",
        },
    }
    result = topic_extractor.extract(event)
    assert result == [], "noreply newsletter email should be skipped"


# ---------------------------------------------------------------------------
# Genuine human emails — should be processed
# ---------------------------------------------------------------------------

def test_genuine_inbound_email_processed(topic_extractor):
    """Inbound email from a real human sender must produce topic signals."""
    event = {
        "type": "email.received",
        "timestamp": "2026-02-28T10:00:00Z",
        "payload": {
            "from": "alice@example.com",
            "subject": "Python code review",
            "body": (
                "Hi, I reviewed your Python code. The machine learning model "
                "looks good but the database query needs optimization. "
                "Let's discuss the architecture tomorrow."
            ),
        },
    }
    result = topic_extractor.extract(event)
    assert len(result) == 1, "Genuine email should produce exactly one topic signal"
    topics = result[0]["topics"]
    assert len(topics) > 0, "Genuine email should produce non-empty topic list"
    # Verify real content words were extracted, not noise
    assert any(t in topics for t in ("python", "machine", "learning", "database", "query", "code", "architecture")), (
        f"Expected technical topics but got: {topics}"
    )


def test_genuine_inbound_email_no_from_processed(topic_extractor):
    """Inbound email with no from field should not be filtered (fail-open)."""
    event = {
        "type": "email.received",
        "timestamp": "2026-02-28T10:00:00Z",
        "payload": {
            "subject": "Project update",
            "body": (
                "The engineering project is making great progress. "
                "The infrastructure team deployed the new system today. "
                "Testing is scheduled for next week."
            ),
        },
    }
    result = topic_extractor.extract(event)
    # Fail-open: no from address means we can't identify it as marketing, so process it
    assert len(result) == 1, "Email without 'from' should be processed (fail-open)"


# ---------------------------------------------------------------------------
# Outbound emails — always processed (users never send automated mail)
# ---------------------------------------------------------------------------

def test_outbound_email_always_processed(topic_extractor):
    """Outbound email must be processed regardless of content (users never send spam)."""
    event = {
        "type": "email.sent",
        "timestamp": "2026-02-28T10:00:00Z",
        "payload": {
            "to": "alice@example.com",
            "subject": "Python project meeting",
            "body": (
                "Looking forward to discussing the Python architecture tomorrow. "
                "Please bring your notes about the database design and testing strategy."
            ),
        },
    }
    result = topic_extractor.extract(event)
    assert len(result) == 1, "Outbound email should always be processed"
    topics = result[0]["topics"]
    assert len(topics) > 0, "Outbound email should produce topics"


# ---------------------------------------------------------------------------
# Direct messages — always processed
# ---------------------------------------------------------------------------

def test_message_received_always_processed(topic_extractor):
    """Inbound direct messages must always be processed (no automated Signal/iMessage)."""
    event = {
        "type": "message.received",
        "timestamp": "2026-02-28T10:00:00Z",
        "payload": {
            "from": "+15551234567",
            "body": (
                "Python conference next week. Want to pair program "
                "on the machine learning project Saturday?"
            ),
        },
    }
    result = topic_extractor.extract(event)
    assert len(result) == 1, "Direct messages should always be processed"


def test_message_sent_always_processed(topic_extractor):
    """Outbound direct messages must always be processed."""
    event = {
        "type": "message.sent",
        "timestamp": "2026-02-28T10:00:00Z",
        "payload": {
            "to": "+15551234567",
            "body": "Sounds great! Python machine learning Saturday works for me.",
        },
    }
    result = topic_extractor.extract(event)
    assert len(result) == 1, "Outbound messages should always be processed"


# ---------------------------------------------------------------------------
# User commands — always processed
# ---------------------------------------------------------------------------

def test_user_command_always_processed(topic_extractor):
    """User voice/text commands must always be processed for topic extraction."""
    event = {
        "type": "system.user.command",
        "timestamp": "2026-02-28T10:00:00Z",
        "payload": {
            "body": "Search for Python machine learning tutorials",
        },
    }
    result = topic_extractor.extract(event)
    assert len(result) == 1, "User commands should always be processed"


# ---------------------------------------------------------------------------
# Content quality: marketing words should NOT appear in topics from genuine emails
# ---------------------------------------------------------------------------

def test_genuine_email_does_not_produce_marketing_topics(topic_extractor):
    """Topics extracted from genuine emails must not contain common marketing words."""
    MARKETING_NOISE = {"offer", "shop", "holiday", "sale", "deal", "store", "email", "reward"}
    event = {
        "type": "email.received",
        "timestamp": "2026-02-28T10:00:00Z",
        "payload": {
            "from": "colleague@company.com",
            "subject": "Q4 engineering roadmap",
            "body": (
                "The engineering roadmap for Q4 focuses on infrastructure scaling, "
                "Python migration, and database performance. The machine learning "
                "team will also be presenting their recommendation system."
            ),
        },
    }
    result = topic_extractor.extract(event)
    assert len(result) == 1
    topics = set(result[0]["topics"])
    noise_found = topics & MARKETING_NOISE
    assert not noise_found, (
        f"Genuine email topics should not contain marketing words, but found: {noise_found}"
    )


# ---------------------------------------------------------------------------
# can_process: verify the event type gating
# ---------------------------------------------------------------------------

def test_can_process_accepts_correct_event_types(topic_extractor):
    """can_process should accept all communication event types."""
    accepted = [
        "email.received", "email.sent",
        "message.received", "message.sent",
        "system.user.command",
    ]
    for event_type in accepted:
        assert topic_extractor.can_process({"type": event_type}), (
            f"can_process should return True for {event_type!r}"
        )


def test_can_process_rejects_non_communication_events(topic_extractor):
    """can_process should reject calendar and system events."""
    rejected = [
        "calendar.event.created",
        "task.created",
        "system.rule.triggered",
        "usermodel.fact.learned",
    ]
    for event_type in rejected:
        assert not topic_extractor.can_process({"type": event_type}), (
            f"can_process should return False for {event_type!r}"
        )
