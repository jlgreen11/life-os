"""
Tests for HTML/CSS token filtering in semantic fact inference.

Verifies that the semantic fact inferrer correctly filters out HTML entities,
CSS properties, and HTML tags from topic-based expertise/interest facts,
preventing semantic memory pollution.
"""

import pytest
from services.semantic_fact_inferrer.inferrer import SemanticFactInferrer


def _set_samples(ums, profile_type, count):
    """Helper to manually set samples_count for a profile."""
    with ums.db.get_connection("user_model") as conn:
        conn.execute(
            "UPDATE signal_profiles SET samples_count = ? WHERE profile_type = ?",
            (count, profile_type)
        )


class TestSemanticHTMLFiltering:
    """Test suite for HTML/CSS token filtering in semantic fact inference."""

    def test_filters_html_entities(self, user_model_store):
        """HTML entities like nbsp, zwnj should not become expertise facts."""
        inferrer = SemanticFactInferrer(user_model_store)

        # Populate topic profile with HTML entity tokens (50+ samples to trigger inference)
        topic_data = {
            "topic_counts": {
                "nbsp": 150,      # Very common HTML entity
                "zwnj": 100,      # Zero-width non-joiner
                "mdash": 80,      # Em dash entity
                "python": 120,    # Legitimate topic (should NOT be filtered)
                "database": 60,   # Legitimate topic (should NOT be filtered)
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics",500)

        # Run inference
        inferrer.infer_from_topic_profile()

        # Verify HTML entities are filtered out
        facts = user_model_store.get_semantic_facts()
        expertise_keys = [f["key"] for f in facts if f["category"] == "expertise"]

        assert "expertise_nbsp" not in expertise_keys
        assert "expertise_zwnj" not in expertise_keys
        assert "expertise_mdash" not in expertise_keys

        # Verify legitimate topics are preserved
        assert "expertise_python" in expertise_keys
        assert "expertise_database" in expertise_keys

    def test_filters_css_properties(self, user_model_store):
        """CSS properties like padding, margin should not become expertise facts."""
        inferrer = SemanticFactInferrer(user_model_store)

        # Populate topic profile with CSS property tokens
        topic_data = {
            "topic_counts": {
                "padding": 200,
                "margin": 180,
                "border": 150,
                "width": 140,
                "height": 130,
                "color": 120,
                "font": 110,
                "javascript": 100,  # Legitimate topic
                "react": 80,        # Legitimate topic
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics",600)

        # Run inference
        inferrer.infer_from_topic_profile()

        # Verify CSS properties are filtered out
        facts = user_model_store.get_semantic_facts()
        expertise_keys = [f["key"] for f in facts if f["category"] == "expertise"]

        assert "expertise_padding" not in expertise_keys
        assert "expertise_margin" not in expertise_keys
        assert "expertise_border" not in expertise_keys
        assert "expertise_width" not in expertise_keys
        assert "expertise_height" not in expertise_keys
        assert "expertise_color" not in expertise_keys
        assert "expertise_font" not in expertise_keys

        # Verify legitimate topics are preserved
        assert "expertise_javascript" in expertise_keys
        assert "expertise_react" in expertise_keys

    def test_filters_html_tags(self, user_model_store):
        """HTML tags like table, tbody should not become expertise facts."""
        inferrer = SemanticFactInferrer(user_model_store)

        # Populate topic profile with HTML tag tokens
        topic_data = {
            "topic_counts": {
                "table": 250,
                "tbody": 200,
                "div": 180,
                "span": 160,
                "img": 140,
                "href": 120,
                "class": 110,
                "kubernetes": 130,  # Legitimate topic
                "docker": 90,       # Legitimate topic
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics",700)

        # Run inference
        inferrer.infer_from_topic_profile()

        # Verify HTML tags are filtered out
        facts = user_model_store.get_semantic_facts()
        expertise_keys = [f["key"] for f in facts if f["category"] == "expertise"]

        assert "expertise_table" not in expertise_keys
        assert "expertise_tbody" not in expertise_keys
        assert "expertise_div" not in expertise_keys
        assert "expertise_span" not in expertise_keys
        assert "expertise_img" not in expertise_keys
        assert "expertise_href" not in expertise_keys
        assert "expertise_class" not in expertise_keys

        # Verify legitimate topics are preserved
        assert "expertise_kubernetes" in expertise_keys
        assert "expertise_docker" in expertise_keys

    def test_filters_css_keywords(self, user_model_store):
        """CSS keywords like important, center should not become expertise facts."""
        inferrer = SemanticFactInferrer(user_model_store)

        # Populate topic profile with CSS keyword tokens
        topic_data = {
            "topic_counts": {
                "important": 180,
                "center": 150,
                "inherit": 120,
                "auto": 100,
                "hidden": 90,
                "machine-learning": 140,  # Legitimate topic
                "artificial-intelligence": 110,  # Legitimate topic
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics",500)

        # Run inference
        inferrer.infer_from_topic_profile()

        # Verify CSS keywords are filtered out
        facts = user_model_store.get_semantic_facts()
        expertise_keys = [f["key"] for f in facts if f["category"] == "expertise"]

        assert "expertise_important" not in expertise_keys
        assert "expertise_center" not in expertise_keys
        assert "expertise_inherit" not in expertise_keys
        assert "expertise_auto" not in expertise_keys
        assert "expertise_hidden" not in expertise_keys

        # Verify legitimate topics are preserved
        assert "expertise_machine-learning" in expertise_keys
        assert "expertise_artificial-intelligence" in expertise_keys

    def test_filters_url_fragments(self, user_model_store):
        """URL fragments like http, https, www should not become expertise facts."""
        inferrer = SemanticFactInferrer(user_model_store)

        # Populate topic profile with URL fragment tokens
        topic_data = {
            "topic_counts": {
                "https": 300,
                "http": 250,
                "www": 200,
                "com": 180,
                "html": 150,
                "css": 130,
                "security": 160,  # Legitimate topic
                "encryption": 110,  # Legitimate topic
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics",800)

        # Run inference
        inferrer.infer_from_topic_profile()

        # Verify URL fragments are filtered out
        facts = user_model_store.get_semantic_facts()
        expertise_keys = [f["key"] for f in facts if f["category"] == "expertise"]

        assert "expertise_https" not in expertise_keys
        assert "expertise_http" not in expertise_keys
        assert "expertise_www" not in expertise_keys
        assert "expertise_com" not in expertise_keys
        assert "expertise_html" not in expertise_keys
        assert "expertise_css" not in expertise_keys

        # Verify legitimate topics are preserved
        assert "expertise_security" in expertise_keys
        assert "expertise_encryption" in expertise_keys

    def test_filters_email_template_artifacts(self, user_model_store):
        """Email template artifacts like unsubscribe, pixel should not become expertise facts."""
        inferrer = SemanticFactInferrer(user_model_store)

        # Populate topic profile with email template artifact tokens
        topic_data = {
            "topic_counts": {
                "unsubscribe": 220,
                "pixel": 180,
                "tracker": 160,
                "analytics": 140,
                "campaign": 120,
                "utm": 100,
                "marketing": 150,  # Legitimate topic (yes, marketing can be a real interest)
                "automation": 110,  # Legitimate topic
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics",600)

        # Run inference
        inferrer.infer_from_topic_profile()

        # Verify email artifacts are filtered out
        facts = user_model_store.get_semantic_facts()
        expertise_keys = [f["key"] for f in facts if f["category"] == "expertise"]

        assert "expertise_unsubscribe" not in expertise_keys
        assert "expertise_pixel" not in expertise_keys
        assert "expertise_tracker" not in expertise_keys
        assert "expertise_analytics" not in expertise_keys
        assert "expertise_campaign" not in expertise_keys
        assert "expertise_utm" not in expertise_keys

        # Verify legitimate topics are preserved
        assert "expertise_marketing" in expertise_keys
        assert "expertise_automation" in expertise_keys

    def test_case_insensitive_filtering(self, user_model_store):
        """Filtering should be case-insensitive (NBSP, Nbsp, nbsp all filtered)."""
        inferrer = SemanticFactInferrer(user_model_store)

        # Populate topic profile with mixed-case HTML/CSS tokens
        topic_data = {
            "topic_counts": {
                "NBSP": 150,      # Uppercase
                "Padding": 140,   # Title case
                "TABLE": 130,     # Uppercase
                "Python": 120,    # Legitimate topic with capital
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics",400)

        # Run inference
        inferrer.infer_from_topic_profile()

        # Verify case-insensitive filtering
        facts = user_model_store.get_semantic_facts()
        expertise_keys = [f["key"] for f in facts if f["category"] == "expertise"]

        # HTML/CSS tokens should be filtered regardless of case
        assert "expertise_NBSP" not in expertise_keys
        assert "expertise_Padding" not in expertise_keys
        assert "expertise_TABLE" not in expertise_keys

        # Legitimate topic should be preserved (exact case from topic_counts key)
        assert "expertise_Python" in expertise_keys

    def test_filters_interest_facts_too(self, user_model_store):
        """HTML/CSS filtering applies to interest facts, not just expertise."""
        inferrer = SemanticFactInferrer(user_model_store)

        # Populate topic profile with tokens that meet interest threshold but not expertise
        # (5+ count, 5%+ frequency but <10 count or <10% frequency)
        topic_data = {
            "topic_counts": {
                "align": 8,       # HTML/CSS token (should be filtered)
                "style": 7,       # HTML/CSS token (should be filtered)
                "blockchain": 9,  # Legitimate topic (should create interest fact)
                "webdev": 6,      # Legitimate topic (should create interest fact)
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics",100)

        # Run inference
        inferrer.infer_from_topic_profile()

        # Verify HTML/CSS tokens don't create interest facts
        facts = user_model_store.get_semantic_facts()
        interest_keys = [f["key"] for f in facts if f["category"] == "implicit_preference"]

        assert "interest_align" not in interest_keys
        assert "interest_style" not in interest_keys

        # Verify legitimate topics create interest facts
        assert "interest_blockchain" in interest_keys
        assert "interest_webdev" in interest_keys

    def test_comprehensive_blocklist_coverage(self, user_model_store):
        """Verify all major HTML/CSS token categories are blocked."""
        inferrer = SemanticFactInferrer(user_model_store)

        # Create a comprehensive test set covering all blocklist categories
        topic_data = {
            "topic_counts": {
                # HTML entities (8 examples)
                "nbsp": 150, "zwnj": 140, "zwj": 130, "mdash": 120,
                "ndash": 110, "hellip": 100, "quot": 90, "amp": 80,

                # CSS properties (10 examples)
                "padding": 200, "margin": 190, "border": 180, "width": 170,
                "height": 160, "color": 150, "font": 140, "align": 130,
                "text": 120, "display": 110,

                # HTML tags (10 examples)
                "table": 250, "tbody": 240, "div": 230, "span": 220,
                "img": 210, "href": 200, "class": 190, "id": 180,
                "meta": 170, "link": 160,

                # CSS keywords (5 examples)
                "important": 180, "center": 170, "inherit": 160,
                "auto": 150, "hidden": 140,

                # URL fragments (5 examples)
                "https": 300, "http": 290, "www": 280, "com": 270, "html": 260,

                # Email artifacts (5 examples)
                "unsubscribe": 220, "pixel": 210, "tracker": 200,
                "campaign": 190, "utm": 180,

                # Legitimate topics (should NOT be filtered)
                "python": 500,
                "javascript": 400,
                "database": 300,
                "api": 200,
                "testing": 150,
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics",1000)

        # Run inference
        inferrer.infer_from_topic_profile()

        # Verify ALL HTML/CSS tokens are filtered
        facts = user_model_store.get_semantic_facts()
        all_keys = [f["key"] for f in facts]

        # Check a representative sample from each category
        assert "expertise_nbsp" not in all_keys
        assert "expertise_padding" not in all_keys
        assert "expertise_table" not in all_keys
        assert "expertise_important" not in all_keys
        assert "expertise_https" not in all_keys
        assert "expertise_unsubscribe" not in all_keys

        # Verify ALL legitimate topics are preserved
        assert "expertise_python" in all_keys
        assert "expertise_javascript" in all_keys
        assert "expertise_database" in all_keys
        assert "expertise_api" in all_keys
        assert "expertise_testing" in all_keys

    def test_empty_topic_profile_no_crash(self, user_model_store):
        """Empty topic profile should not crash when filtering."""
        inferrer = SemanticFactInferrer(user_model_store)

        # Empty topic profile
        topic_data = {"topic_counts": {}}
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics",100)

        # Should not crash
        inferrer.infer_from_topic_profile()

        # No facts should be created
        facts = user_model_store.get_semantic_facts()
        assert len(facts) == 0

    def test_all_filtered_topics_no_crash(self, user_model_store):
        """Profile with only HTML/CSS tokens should not crash."""
        inferrer = SemanticFactInferrer(user_model_store)

        # Only HTML/CSS tokens (all should be filtered)
        topic_data = {
            "topic_counts": {
                "nbsp": 200,
                "padding": 180,
                "table": 160,
                "https": 140,
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics",400)

        # Should not crash
        inferrer.infer_from_topic_profile()

        # No expertise/interest facts should be created
        facts = user_model_store.get_semantic_facts()
        expertise_interest_facts = [
            f for f in facts
            if f["category"] in ("expertise", "implicit_preference")
        ]
        assert len(expertise_interest_facts) == 0

    def test_logging_reports_filtered_count(self, user_model_store, caplog):
        """Verify that filtering events are logged for visibility."""
        import logging
        caplog.set_level(logging.INFO)

        inferrer = SemanticFactInferrer(user_model_store)

        # Mix of filtered and legitimate tokens
        topic_data = {
            "topic_counts": {
                "nbsp": 150,
                "padding": 140,
                "table": 130,
                "python": 120,
            }
        }
        user_model_store.update_signal_profile("topics", topic_data)
        _set_samples(user_model_store, "topics",400)

        # Run inference
        inferrer.infer_from_topic_profile()

        # Verify logging message reports filtered count
        assert any("Filtered 3 HTML/CSS tokens" in record.message for record in caplog.records)
