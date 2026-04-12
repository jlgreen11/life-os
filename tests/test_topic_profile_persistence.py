"""
Tests for topic extractor profile persistence and HTML parsing safety.

Covers:
- extract() persists the topics profile to user_model.db
- Processed HTML email bodies yield correct topic_counts entries
- Malformed HTML falls back to regex stripping and still produces topics
- Topics extracted from HTML bodies do NOT include HTML/CSS garbage tokens
- post-write verification path (error logged when profile unreadable)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from unittest.mock import patch

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
    """Helper: build a minimal email.received event dict."""
    return {
        "id": "test-event-1",
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
# Profile persistence
# ---------------------------------------------------------------------------


class TestTopicProfilePersistence:
    """Verify that extract() actually writes the topics profile to the DB."""

    def test_extract_creates_topics_profile(self, topic_extractor):
        """After extracting from an email, the topics signal profile must exist."""
        event = _email_event(
            subject="Python software engineering",
            body=(
                "This email discusses Python software engineering practices "
                "including testing, refactoring, and performance optimization."
            ),
        )

        signals = topic_extractor.extract(event)

        # At least one signal should be produced
        assert len(signals) == 1, "Expected one topic signal"

        # The profile must now be readable from the DB
        profile = topic_extractor.ums.get_signal_profile("topics")
        assert profile is not None, "topics profile must exist in user_model.db after extraction"
        assert "data" in profile

        data = profile["data"]
        assert "topic_counts" in data, "profile data must contain topic_counts"
        assert "recent_topics" in data, "profile data must contain recent_topics"
        assert len(data["topic_counts"]) > 0, "topic_counts must be non-empty"
        assert len(data["recent_topics"]) == 1, "recent_topics must have exactly one entry"

    def test_extract_html_body_creates_profile_with_expected_keywords(self, topic_extractor):
        """HTML email body should yield topic_counts with meaningful keywords."""
        html_body = """
        <html>
        <head><style>.btn { color: blue; padding: 10px; }</style></head>
        <body>
            <p>Welcome to our <strong>machine learning</strong> workshop!</p>
            <p>Topics covered: neural networks, deep learning, data science,
               model evaluation and optimization techniques.</p>
        </body>
        </html>
        """
        event = _email_event(
            subject="Machine Learning Workshop",
            body=html_body,
        )

        signals = topic_extractor.extract(event)
        assert len(signals) == 1

        profile = topic_extractor.ums.get_signal_profile("topics")
        assert profile is not None

        topic_counts = profile["data"]["topic_counts"]

        # These content keywords should be present
        content_keywords = {"machine", "learning", "neural", "networks", "data", "science"}
        found = content_keywords & set(topic_counts.keys())
        assert found, (
            f"Expected at least one of {content_keywords} in topic_counts, "
            f"got: {set(topic_counts.keys())}"
        )

        # HTML/CSS tokens must NOT be present
        html_css_garbage = {"html", "head", "body", "style", "padding", "color", "blue", "nbsp", "btn"}
        leaked = html_css_garbage & set(topic_counts.keys())
        assert not leaked, f"HTML/CSS tokens leaked into topic_counts: {leaked}"

    def test_multiple_extractions_accumulate_counts(self, topic_extractor):
        """Successive extractions should increment topic_counts correctly."""
        for i in range(3):
            topic_extractor.extract(
                _email_event(
                    body=f"Email {i} discussing Python software testing and deployment strategies.",
                )
            )

        profile = topic_extractor.ums.get_signal_profile("topics")
        counts = profile["data"]["topic_counts"]

        # "python" appears in all 3 messages — count must be 3
        assert counts.get("python", 0) == 3, (
            f"Expected python count=3, got {counts.get('python', 0)}"
        )
        # recent_topics ring must have 3 entries
        assert len(profile["data"]["recent_topics"]) == 3

    def test_topic_counts_contain_extracted_signal_keywords(self, topic_extractor):
        """Keywords returned in the signal must also be present in topic_counts."""
        event = _email_event(
            body=(
                "Kubernetes container orchestration simplifies deployment pipelines "
                "for microservices architecture and cloud infrastructure management."
            )
        )

        signals = topic_extractor.extract(event)
        assert len(signals) == 1

        extracted_topics = set(signals[0]["topics"])
        profile = topic_extractor.ums.get_signal_profile("topics")
        persisted_counts = set(profile["data"]["topic_counts"].keys())

        # Every keyword returned in the signal must be persisted
        missing = extracted_topics - persisted_counts
        assert not missing, (
            f"These keywords were in the signal but not persisted: {missing}"
        )


# ---------------------------------------------------------------------------
# HTML parsing safety
# ---------------------------------------------------------------------------


class TestHTMLParsingSafety:
    """Verify the HTML parsing fallback handles malformed markup gracefully."""

    def test_malformed_html_unclosed_tags_still_extracts_topics(self, topic_extractor):
        """Unclosed tags must not crash extraction — fallback produces usable text."""
        malformed = (
            "<p>This email discusses machine learning and artificial intelligence "
            "<b>without closing tags"
        )
        event = _email_event(body=malformed)

        # Should not raise
        signals = topic_extractor.extract(event)
        assert len(signals) == 1

        topics = signals[0]["topics"]
        assert any(kw in topics for kw in ["machine", "learning", "artificial", "intelligence"]), (
            f"Expected ML keywords from malformed HTML, got: {topics}"
        )

    def test_malformed_html_invalid_entities_logs_warning_and_fallsback(self, topic_extractor, caplog):
        """
        When HTMLParser raises on truly broken markup, a warning is logged and
        the regex fallback is used so processing continues.
        """
        # Force HTMLStripper.feed() to raise by patching it
        original_extract = topic_extractor._strip_html

        def raising_stripper(html_text: str) -> str:
            """Simulate HTMLParser choking on corrupted input."""
            from services.signal_extractor.topic import HTMLStripper
            with patch.object(HTMLStripper, "feed", side_effect=Exception("encoding error")):
                return original_extract(html_text)

        topic_extractor._strip_html = raising_stripper

        html_body = (
            "<p>This email discusses Python programming language, "
            "software testing and continuous integration practices.</p>"
        )
        event = _email_event(body=html_body)

        with caplog.at_level(logging.WARNING, logger="services.signal_extractor.topic"):
            signals = topic_extractor.extract(event)

        # Warning must be logged
        warning_messages = [r.message for r in caplog.records if r.levelno == logging.WARNING]
        assert any("HTML parsing failed" in m for m in warning_messages), (
            f"Expected 'HTML parsing failed' warning, got: {warning_messages}"
        )

        # Extraction should still yield topics despite the parsing error
        assert len(signals) == 1
        topics = signals[0]["topics"]
        assert any(kw in topics for kw in ["python", "programming", "testing", "integration"]), (
            f"Expected meaningful topics from regex fallback, got: {topics}"
        )

    def test_strip_html_regex_fallback_removes_tags(self, topic_extractor):
        """The regex fallback path in _strip_html must strip tags correctly."""
        from services.signal_extractor.topic import HTMLStripper

        with patch.object(HTMLStripper, "feed", side_effect=Exception("bad html")):
            result = topic_extractor._strip_html(
                "<div>Hello <span>world</span> content</div>"
            )

        assert "<" not in result, "All tags must be removed by the regex fallback"
        assert "Hello" in result
        assert "world" in result
        assert "content" in result

    def test_deeply_nested_html_extracts_inner_text(self, topic_extractor):
        """Deeply nested legitimate HTML should still yield meaningful text."""
        deep_html = """
        <html><body>
          <table><tbody><tr><td>
            <div><p><span>
              Kubernetes orchestration simplifies container deployment
              for distributed systems and microservices.
            </span></p></div>
          </td></tr></tbody></table>
        </body></html>
        """
        event = _email_event(body=deep_html)
        signals = topic_extractor.extract(event)

        assert len(signals) == 1
        topics = signals[0]["topics"]

        # Structural tokens must not appear
        structure_tokens = {"table", "tbody", "span", "html", "body"}
        leaked = structure_tokens & set(topics)
        assert not leaked, f"HTML structure tokens leaked: {leaked}"

        # Content keywords must be present
        assert any(kw in topics for kw in ["kubernetes", "container", "deployment"]), (
            f"Expected deployment-related keywords, got: {topics}"
        )


# ---------------------------------------------------------------------------
# Post-write verification
# ---------------------------------------------------------------------------


class TestPostWriteVerification:
    """Verify the post-write check logs an error when profile is unreadable."""

    def test_error_logged_when_profile_unreadable_after_write(self, topic_extractor, caplog):
        """If get_signal_profile returns None after a write, an error must be logged."""
        event = _email_event(
            body="Python software development testing and deployment automation practices.",
        )

        # Patch get_signal_profile to simulate a DB read failure after the write
        original_get = topic_extractor.ums.get_signal_profile

        call_count = {"n": 0}

        def failing_get(profile_type: str):
            """Return None on the verification call (second call per update)."""
            call_count["n"] += 1
            # The first call in _update_topic_map reads existing data (returns None
            # when there is no prior profile — that is normal).  The second call is
            # the post-write verification; we fail that one to trigger the error log.
            if call_count["n"] >= 2:
                return None
            return original_get(profile_type)

        topic_extractor.ums.get_signal_profile = failing_get

        with caplog.at_level(logging.ERROR, logger="services.signal_extractor.topic"):
            topic_extractor.extract(event)

        error_messages = [r.message for r in caplog.records if r.levelno == logging.ERROR]
        assert any("FAILED to persist" in m for m in error_messages), (
            f"Expected 'FAILED to persist' error log, got: {error_messages}"
        )
