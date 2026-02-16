"""
Tests for HTML stripping in topic extraction.

These tests verify that HTML email bodies are properly cleaned before topic
extraction, preventing HTML/CSS tokens from polluting semantic memory.
"""

from datetime import datetime, timezone

import pytest

from models.core import EventType
from services.signal_extractor.topic import TopicExtractor


@pytest.fixture
def topic_extractor(db, user_model_store):
    """Create a TopicExtractor instance with a test database and user model store."""
    return TopicExtractor(db=db, user_model_store=user_model_store)


class TestHTMLDetection:
    """Test the HTML detection heuristic."""

    def test_detects_simple_html(self, topic_extractor):
        """Should detect basic HTML tags."""
        assert topic_extractor._is_html("<p>Hello</p>")
        assert topic_extractor._is_html("<div>Content</div>")
        assert topic_extractor._is_html("<table><tr><td>Data</td></tr></table>")

    def test_detects_html_with_attributes(self, topic_extractor):
        """Should detect HTML tags with attributes."""
        assert topic_extractor._is_html('<p class="text">Hello</p>')
        assert topic_extractor._is_html('<div style="color: red;">Content</div>')
        assert topic_extractor._is_html('<a href="http://example.com">Link</a>')

    def test_does_not_detect_plain_text(self, topic_extractor):
        """Should not detect plain text as HTML."""
        assert not topic_extractor._is_html("This is plain text")
        assert not topic_extractor._is_html("No tags here!")
        assert not topic_extractor._is_html("Math: 5 < 10 and 20 > 15")

    def test_handles_edge_cases(self, topic_extractor):
        """Should handle edge cases correctly."""
        # Empty string
        assert not topic_extractor._is_html("")
        # Just angle brackets (not tags)
        assert not topic_extractor._is_html("< >")
        # Email addresses with angle brackets
        assert not topic_extractor._is_html("Contact: <user@example.com>")


class TestHTMLStripping:
    """Test HTML stripping functionality."""

    def test_strips_simple_tags(self, topic_extractor):
        """Should strip basic HTML tags."""
        html = "<p>Hello world</p>"
        result = topic_extractor._strip_html(html)
        assert result == "Hello world"

    def test_strips_nested_tags(self, topic_extractor):
        """Should strip nested HTML tags."""
        html = "<div><p>Hello <b>world</b>!</p></div>"
        result = topic_extractor._strip_html(html)
        # HTMLParser preserves spacing between elements
        assert "Hello" in result and "world" in result
        assert "!" in result

    def test_strips_table_markup(self, topic_extractor):
        """Should strip table markup that caused the HTML token pollution."""
        html = """
        <table border="1" cellpadding="5">
            <tbody>
                <tr>
                    <td width="100">Name</td>
                    <td height="20">Value</td>
                </tr>
            </tbody>
        </table>
        """
        result = topic_extractor._strip_html(html)
        # Should extract only the visible text, not HTML tokens like
        # 'border', 'tbody', 'cellpadding', 'width', 'height'
        assert "Name" in result
        assert "Value" in result
        assert "tbody" not in result.lower()
        assert "cellpadding" not in result.lower()
        assert "border" not in result.lower()

    def test_decodes_html_entities(self, topic_extractor):
        """Should decode HTML entities like &nbsp;."""
        html = "Order total: $50&nbsp;USD"
        result = topic_extractor._strip_html(html)
        # &nbsp; should be decoded to a space
        assert "USD" in result
        assert "&nbsp;" not in result
        # Check that we can extract meaningful tokens (not 'nbsp')
        assert "nbsp" not in result.lower()

    def test_handles_malformed_html(self, topic_extractor):
        """Should gracefully handle malformed HTML."""
        # Unclosed tags
        html = "<p>Hello <b>world"
        result = topic_extractor._strip_html(html)
        assert "Hello" in result
        assert "world" in result

        # Invalid nesting
        html = "<p><div>Content</p></div>"
        result = topic_extractor._strip_html(html)
        assert "Content" in result

    def test_preserves_text_spacing(self, topic_extractor):
        """Should preserve word boundaries when stripping HTML."""
        html = "<p>First paragraph.</p><p>Second paragraph.</p>"
        result = topic_extractor._strip_html(html)
        # Words should be separated by spaces, not concatenated
        assert "paragraph" in result
        # Should not have "paragraph.Second"
        words = result.split()
        assert "paragraph.Second" not in words

    def test_normalizes_whitespace(self, topic_extractor):
        """Should normalize excessive whitespace."""
        html = """
        <div>
            Multiple    spaces   and
            newlines   everywhere
        </div>
        """
        result = topic_extractor._strip_html(html)
        # Should collapse to single spaces
        assert "  " not in result
        assert "\n" not in result


class TestTopicExtractionWithHTML:
    """Test topic extraction from HTML emails end-to-end."""

    def test_html_email_extracts_content_not_markup(self, topic_extractor):
        """HTML email should extract content topics, not HTML/CSS tokens."""
        event = {
            "id": "evt_html_email",
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "subject": "Python Programming Newsletter",
                "body": """
                <html>
                <head><style>.header { color: blue; padding: 10px; }</style></head>
                <body>
                    <table border="1" cellpadding="5">
                        <tbody>
                            <tr>
                                <td width="200">
                                    <p>Learn <b>Python</b> programming with our tutorial!</p>
                                    <p>Topics include: functions, classes, modules, testing.</p>
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </body>
                </html>
                """
            }
        }

        signals = topic_extractor.extract(event)

        assert len(signals) == 1
        topics = signals[0]["topics"]

        # Should extract content topics
        assert "python" in topics or "programming" in topics or "tutorial" in topics

        # Should NOT extract HTML/CSS tokens that caused the pollution
        html_css_tokens = {
            "html", "head", "body", "table", "tbody", "border",
            "cellpadding", "width", "style", "header", "color",
            "padding", "blue", "nbsp"
        }
        extracted_tokens = set(t.lower() for t in topics)
        pollution = extracted_tokens & html_css_tokens

        assert not pollution, f"HTML/CSS tokens leaked into topics: {pollution}"

    def test_plain_text_email_unchanged(self, topic_extractor):
        """Plain text emails should work exactly as before."""
        event = {
            "id": "evt_plain_email",
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "subject": "Project Update",
                "body": "The project milestone has been completed successfully. "
                        "We implemented the feature using Python and tested thoroughly."
            }
        }

        signals = topic_extractor.extract(event)

        assert len(signals) == 1
        topics = signals[0]["topics"]

        # Should extract meaningful topics
        content_words = {"project", "milestone", "completed", "feature", "python", "tested"}
        assert any(word in topics for word in content_words)

    def test_marketing_email_with_heavy_html(self, topic_extractor):
        """Marketing emails with heavy HTML should extract content, not formatting."""
        event = {
            "id": "evt_marketing",
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "subject": "Special Offer",
                "body": """
                <html>
                <body style="font-family: Arial; background-color: white;">
                    <table width="600" cellpadding="0" cellspacing="0" border="0">
                        <tr>
                            <td align="center" valign="top">
                                <p style="font-size: 18px; font-weight: bold; color: #333;">
                                    Save 50% on premium membership today!
                                </p>
                                <p style="font-size: 14px; color: #666;">
                                    Limited time offer expires soon.
                                </p>
                            </td>
                        </tr>
                    </table>
                </body>
                </html>
                """
            }
        }

        signals = topic_extractor.extract(event)

        if signals:  # May be filtered by length check
            topics = signals[0]["topics"]

            # Should NOT include formatting tokens
            formatting_tokens = {
                "arial", "font", "size", "weight", "bold", "color",
                "align", "center", "valign", "cellpadding", "cellspacing"
            }
            extracted_tokens = set(t.lower() for t in topics)
            pollution = extracted_tokens & formatting_tokens

            assert not pollution, f"Formatting tokens leaked into topics: {pollution}"

    def test_email_with_entities(self, topic_extractor):
        """Email with HTML entities in HTML markup should decode them correctly."""
        event = {
            "id": "evt_entities",
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "subject": "Invoice",
                "body": "<p>Order total: $500&nbsp;USD.</p><p>Payment&nbsp;due&nbsp;immediately.</p>"
            }
        }

        signals = topic_extractor.extract(event)

        assert len(signals) == 1
        topics = signals[0]["topics"]

        # Should NOT include 'nbsp' as a topic
        assert "nbsp" not in [t.lower() for t in topics]

        # Should extract actual content
        content_words = {"order", "total", "payment"}
        assert any(word in [t.lower() for t in topics] for word in content_words)


class TestBackwardsCompatibility:
    """Test that the fix doesn't break existing functionality."""

    def test_body_plain_still_works(self, topic_extractor):
        """Should still process body_plain when available."""
        event = {
            "id": "evt_plain_body",
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "subject": "Meeting",
                "body_plain": "Let's discuss the Python project tomorrow."
            }
        }

        signals = topic_extractor.extract(event)

        assert len(signals) == 1
        topics = signals[0]["topics"]
        assert "python" in [t.lower() for t in topics] or "project" in [t.lower() for t in topics]

    def test_subject_only_emails(self, topic_extractor):
        """Should extract topics from subject-only emails."""
        event = {
            "id": "evt_subject_only",
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "subject": "Python conference registration confirmation",
                "body": ""
            }
        }

        signals = topic_extractor.extract(event)

        assert len(signals) == 1
        topics = signals[0]["topics"]
        assert "python" in [t.lower() for t in topics] or "conference" in [t.lower() for t in topics]

    def test_short_messages_still_filtered(self, topic_extractor):
        """Should still skip messages that are too short."""
        event = {
            "id": "evt_short",
            "type": EventType.EMAIL_RECEIVED.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "payload": {
                "subject": "",
                "body": "OK"
            }
        }

        signals = topic_extractor.extract(event)

        # Should be filtered out due to length
        assert len(signals) == 0
