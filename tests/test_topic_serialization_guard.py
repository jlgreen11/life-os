"""
Tests for the JSON serialization guard in TopicExtractor._update_topic_map().

The guard was added because update_signal_profile() silently swallows
serialization failures inside a broad try/except.  If topic data contains a
non-serializable type (set, datetime, Enum, etc.), the write would fail with no
visible error and the topics profile would remain empty.

Covers:
- Non-serializable value in topic_counts causes write to be skipped and
  an error is logged identifying the bad field
- Non-serializable field in a recent_topics entry causes write to be skipped
  and an error is logged identifying the bad entry/field
- Normal serializable data still writes successfully (guard does not block
  valid data)
- Set value in topic_counts is identified specifically in the error log
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import pytest

from models.core import EventType
from services.signal_extractor.topic import TopicExtractor
from storage.user_model_store import UserModelStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def topic_extractor(db, user_model_store):
    """TopicExtractor wired to a real in-memory UserModelStore."""
    return TopicExtractor(db=db, user_model_store=user_model_store)


def _email_event(body: str, subject: str = "", from_addr: str = "alice@example.com") -> dict:
    """Build a minimal email.received event dict for testing."""
    return {
        "id": "test-event-guard-1",
        "type": EventType.EMAIL_RECEIVED.value,
        "source": "proton_mail",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "payload": {
            "subject": subject,
            "body": body,
            "from": from_addr,
        },
    }


# ---------------------------------------------------------------------------
# Serialization guard tests
# ---------------------------------------------------------------------------


class TestSerializationGuard:
    """Verify the pre-write JSON serialization guard in _update_topic_map()."""

    def test_set_in_topic_counts_skips_write_and_logs_error(
        self, topic_extractor, caplog
    ):
        """
        When topic_counts contains a set() value (non-serializable), the
        guard must skip the DB write and log an error naming the bad field.

        The bad count is placed under a key ('corrupted_topic') that is NOT
        in the signal's topics list, so the increment loop never touches it
        and the TypeError surfaces only at the json.dumps check in the guard.
        """
        bad_signal = {
            "type": "topic",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topics": ["python", "testing"],
            "context": EventType.EMAIL_RECEIVED.value,
            "source": "proton_mail",
        }

        original_get = topic_extractor.ums.get_signal_profile

        def get_with_bad_data(profile_type: str):
            """Return profile data with a set in topic_counts under an unrelated key."""
            existing = original_get(profile_type)
            if existing:
                return existing
            # 'corrupted_topic' is not in signal["topics"], so the + 1 increment
            # loop skips it.  The set value is only caught at json.dumps time.
            return {
                "data": {
                    "topic_counts": {"corrupted_topic": {1, 2, 3}},  # set — not JSON-serializable
                    "recent_topics": [],
                }
            }

        topic_extractor.ums.get_signal_profile = get_with_bad_data

        with caplog.at_level(logging.ERROR, logger="services.signal_extractor.topic"):
            topic_extractor._update_topic_map(bad_signal)

        # Restore original get to check real DB state
        topic_extractor.ums.get_signal_profile = original_get
        profile = topic_extractor.ums.get_signal_profile("topics")
        assert profile is None, (
            "Profile must not be written when serialization guard fires; "
            f"got: {profile}"
        )

        # An error must be logged identifying the serialization failure
        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("non-JSON-serializable" in m for m in error_messages), (
            f"Expected 'non-JSON-serializable' in error log; got: {error_messages}"
        )

    def test_set_in_topic_counts_error_identifies_bad_field(
        self, topic_extractor, caplog
    ):
        """
        The error log must name the specific topic_counts key whose value
        is non-serializable so the root cause is immediately diagnosable.

        The bad count is stored under 'legacy_topic', which is NOT in the
        signal's topics list, so the increment loop never modifies it.
        """
        bad_signal = {
            "type": "topic",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topics": ["machine", "learning"],
            "context": EventType.EMAIL_RECEIVED.value,
            "source": "proton_mail",
        }

        original_get = topic_extractor.ums.get_signal_profile

        def get_with_set_count(profile_type: str):
            existing = original_get(profile_type)
            if existing:
                return existing
            # 'legacy_topic' is not in signal["topics"] so it won't be
            # incremented — the bad set only surfaces at json.dumps time.
            return {
                "data": {
                    "topic_counts": {"legacy_topic": {5, 6}},  # set — not serializable
                    "recent_topics": [],
                }
            }

        topic_extractor.ums.get_signal_profile = get_with_set_count

        with caplog.at_level(logging.ERROR, logger="services.signal_extractor.topic"):
            topic_extractor._update_topic_map(bad_signal)

        # The error log must mention the offending field name
        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("topic_counts" in m and "legacy_topic" in m for m in error_messages), (
            f"Expected error to identify 'topic_counts[legacy_topic]'; got: {error_messages}"
        )

    def test_set_in_recent_topics_entry_skips_write_and_logs_error(
        self, topic_extractor, caplog
    ):
        """
        When a recent_topics entry contains a non-serializable field (e.g.
        a set in the 'topics' list element), the guard must skip the write
        and log an error identifying the bad entry.
        """
        bad_signal = {
            "type": "topic",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topics": ["cloud", "infrastructure"],
            "context": EventType.EMAIL_RECEIVED.value,
            "source": "proton_mail",
        }

        original_get = topic_extractor.ums.get_signal_profile

        def get_with_bad_recent(profile_type: str):
            existing = original_get(profile_type)
            if existing:
                return existing
            return {
                "data": {
                    "topic_counts": {"cloud": 1},
                    "recent_topics": [
                        {
                            "topics": {"cloud", "infrastructure"},  # set, not serializable
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "context": "email.received",
                        }
                    ],
                }
            }

        topic_extractor.ums.get_signal_profile = get_with_bad_recent

        with caplog.at_level(logging.ERROR, logger="services.signal_extractor.topic"):
            topic_extractor._update_topic_map(bad_signal)

        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("non-JSON-serializable" in m for m in error_messages), (
            f"Expected 'non-JSON-serializable' in error log; got: {error_messages}"
        )

        # Verify write was skipped
        topic_extractor.ums.get_signal_profile = original_get
        assert topic_extractor.ums.get_signal_profile("topics") is None

    def test_recent_topics_error_identifies_bad_entry_field(
        self, topic_extractor, caplog
    ):
        """
        The error log must identify which recent_topics entry index and which
        field within that entry is non-serializable.
        """
        bad_signal = {
            "type": "topic",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "topics": ["docker", "containers"],
            "context": EventType.EMAIL_RECEIVED.value,
            "source": "proton_mail",
        }

        original_get = topic_extractor.ums.get_signal_profile

        def get_with_bad_context(profile_type: str):
            existing = original_get(profile_type)
            if existing:
                return existing
            return {
                "data": {
                    "topic_counts": {"docker": 2},
                    "recent_topics": [
                        {
                            "topics": ["docker"],
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "context": {"not", "serializable"},  # set in context field
                        }
                    ],
                }
            }

        topic_extractor.ums.get_signal_profile = get_with_bad_context

        with caplog.at_level(logging.ERROR, logger="services.signal_extractor.topic"):
            topic_extractor._update_topic_map(bad_signal)

        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        # Error should mention recent_topics[0]['context']
        assert any("recent_topics" in m and "context" in m for m in error_messages), (
            f"Expected error to identify 'recent_topics[0][context]'; got: {error_messages}"
        )

    def test_valid_data_writes_successfully(self, topic_extractor, caplog):
        """
        The serialization guard must NOT block writes for fully serializable
        data.  Normal operation must continue to work.
        """
        event = _email_event(
            subject="Python deployment automation",
            body=(
                "Automating Python deployment pipelines with Docker containers "
                "and Kubernetes orchestration simplifies infrastructure management."
            ),
        )

        with caplog.at_level(logging.ERROR, logger="services.signal_extractor.topic"):
            signals = topic_extractor.extract(event)

        # No errors should be logged for valid data
        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        serialization_errors = [m for m in error_messages if "non-JSON-serializable" in m]
        assert not serialization_errors, (
            f"Guard blocked valid data write; errors: {serialization_errors}"
        )

        # Signal should be produced
        assert len(signals) == 1, "Expected one topic signal for valid event"

        # Profile must be written successfully
        profile = topic_extractor.ums.get_signal_profile("topics")
        assert profile is not None, "Profile must be written for valid serializable data"
        assert "topic_counts" in profile["data"]
        assert len(profile["data"]["topic_counts"]) > 0

    def test_guard_does_not_interfere_with_accumulation(self, topic_extractor):
        """
        Multiple consecutive valid extractions must accumulate counts correctly
        — the guard must not interfere with normal accumulation.
        """
        body = "Python testing software deployment automation engineering practices."
        for _ in range(3):
            topic_extractor.extract(_email_event(body=body))

        profile = topic_extractor.ums.get_signal_profile("topics")
        assert profile is not None
        counts = profile["data"]["topic_counts"]

        # "python" appears in all 3 messages
        assert counts.get("python", 0) == 3, (
            f"Expected python count=3, got {counts.get('python', 0)}"
        )
        assert len(profile["data"]["recent_topics"]) == 3
